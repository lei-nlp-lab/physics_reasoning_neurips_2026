import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution instance.
        Uses getattr with fallback to results dictionary.
        Converts all lengths to meters for SI consistency.
        """
        # Inputs (convert cm to m)
        self.l0 = getattr(solution_instance, 'l0', 
                         solution_instance.results.get('l0', 0.15))  # 15 cm = 0.15 m
        self.l1 = getattr(solution_instance, 'l1', 
                         solution_instance.results.get('l1', 0.20))  # 20 cm = 0.20 m
        self.F = getattr(solution_instance, 'F', 
                        solution_instance.results.get('F', 400.0))   # 400 N
        
        # Results (using keys from mapping)
        self.k = getattr(solution_instance, 'k', 
                        solution_instance.results.get('k', 0.0))
        self.W = getattr(solution_instance, 'W', 
                        solution_instance.results.get('W', 0.0))
        # Try multiple possible keys for delta_Ep
        self.delta_Ep = getattr(solution_instance, 'delta_Ep', 
                               solution_instance.results.get('$DeltaE_p$', 
                               solution_instance.results.get('$\Delta E_{p}$', 0.0)))
        
        # Derived quantities
        self.delta_l = self.l1 - self.l0  # elongation in meters
        
        # Constants
        self.expected_k = 8000.0  # N/m from step 1
        self.expected_W = -10.0   # J from step 2
        self.expected_delta_Ep = 10.0  # J from step 3
        
        # For monotonicity test
        self.original_l1 = self.l1

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics laws:
        1. Hooke's Law: F = k * delta_l
        2. Energy consistency: delta_Ep = -W
        3. Work-energy theorem for springs: W = -1/2 * k * delta_l^2
        """
        logs = []
        success = True
        
        # 1. Hooke's Law: F = k * delta_l (using magnitudes, signs handled separately)
        hooke_lhs = self.F
        hooke_rhs = self.k * abs(self.delta_l)
        if not math.isclose(hooke_lhs, hooke_rhs, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Hooke's Law violation. F={hooke_lhs:.2f} != k*Δl={hooke_rhs:.2f}")
            success = False
        else:
            logs.append(f"PASS: Hooke's Law F = k*Δl: {hooke_lhs:.2f} ≈ {hooke_rhs:.2f}")
        
        # 2. Energy consistency: delta_Ep = -W
        if not math.isclose(self.delta_Ep, -self.W, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Energy inconsistency. ΔEp={self.delta_Ep:.2f} != -W={-self.W:.2f}")
            success = False
        else:
            logs.append(f"PASS: ΔEp = -W: {self.delta_Ep:.2f} ≈ {-self.W:.2f}")
        
        # 3. Work formula: W = -1/2 * k * delta_l^2
        W_calc = -0.5 * self.k * (self.delta_l ** 2)
        if not math.isclose(self.W, W_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Work formula mismatch. W={self.W:.2f} != -1/2*k*Δl²={W_calc:.2f}")
            success = False
        else:
            logs.append(f"PASS: W = -1/2*k*Δl²: {self.W:.2f} ≈ {W_calc:.2f}")
        
        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Compare extracted results with expected values from steps analysis.
        """
        logs = []
        success = True
        
        checks = [
            ("k", self.k, self.expected_k, "N/m"),
            ("W", self.W, self.expected_W, "J"),
            ("ΔEp", self.delta_Ep, self.expected_delta_Ep, "J")
        ]
        
        for name, actual, expected, unit in checks:
            if not math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: {name}={actual:.2f}{unit} ≠ expected {expected:.2f}{unit}")
                success = False
            else:
                logs.append(f"PASS: {name}={actual:.2f}{unit} matches expected")
        
        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb elongation by +1%, recompute derived quantities using formulas,
        verify monotonic trends, then restore original state.
        """
        logs = []
        success = True
        
        # Store original values
        original_l1 = self.l1
        original_delta_l = self.delta_l
        original_W = self.W
        original_delta_Ep = self.delta_Ep
        
        try:
            # Perturb: increase elongation by 1%
            perturb_factor = 1.01
            self.l1 = original_l1 * perturb_factor
            self.delta_l = self.l1 - self.l0
            
            # Recompute using formulas (not solution methods)
            # Force should increase linearly (F = k * delta_l)
            F_new = self.k * abs(self.delta_l)
            # Work magnitude should increase quadratically
            W_new = -0.5 * self.k * (self.delta_l ** 2)
            # Potential energy should increase quadratically
            delta_Ep_new = 0.5 * self.k * (self.delta_l ** 2)
            
            # Check monotonic trends
            if abs(self.delta_l) <= abs(original_delta_l):
                logs.append(f"FAIL: |Δl| decreased after perturbation: |{self.delta_l:.4f}| ≤ |{original_delta_l:.4f}|")
                success = False
            else:
                logs.append(f"PASS: |Δl| increased: |{self.delta_l:.4f}| > |{original_delta_l:.4f}|")
            
            if F_new <= self.F:
                logs.append(f"FAIL: Force didn't increase: {F_new:.2f} ≤ {self.F:.2f}")
                success = False
            else:
                logs.append(f"PASS: Force increased: {F_new:.2f} > {self.F:.2f}")
            
            # W is negative, so more negative means magnitude increased
            if W_new >= original_W:
                logs.append(f"FAIL: |W| didn't increase: {W_new:.2f} ≥ {original_W:.2f}")
                success = False
            else:
                logs.append(f"PASS: |W| increased: {W_new:.2f} < {original_W:.2f} (more negative)")
            
            if delta_Ep_new <= original_delta_Ep:
                logs.append(f"FAIL: ΔEp didn't increase: {delta_Ep_new:.2f} ≤ {original_delta_Ep:.2f}")
                success = False
            else:
                logs.append(f"PASS: ΔEp increased: {delta_Ep_new:.2f} > {original_delta_Ep:.2f}")
                
        finally:
            # Restore original state
            self.l1 = original_l1
            self.delta_l = original_delta_l
            # Note: self.W and self.delta_Ep are unchanged as they're from solution
            
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and consolidated logs.
        """
        all_logs = ["=== Physics Audit Report ==="]
        overall_success = True
        
        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("\n1. PHYSICS INVARIANTS:")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 2: Trace consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n2. TRACE CONSISTENCY:")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n3. MONOTONICITY CHECK (+1% perturbation):")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Summary
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs

# Optional: Example usage guard
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'k': 8000.0,
                'W': -10.0,
                '$DeltaE_p$': 10.0,
                'l0': 0.15,
                'l1': 0.20,
                'F': 400.0
            }
    
    mock = MockSolution()
    checker = PhysicsChecker(mock)
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)