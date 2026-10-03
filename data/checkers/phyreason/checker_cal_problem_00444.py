import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs and results
        self.mA = getattr(solution_instance, 'mA', 
                         getattr(solution_instance, 'results', {}).get('mA', 5.0))
        self.mB = getattr(solution_instance, 'mB', 
                         getattr(solution_instance, 'results', {}).get('mB', 3.0))
        self.F = getattr(solution_instance, 'F', 
                        getattr(solution_instance, 'results', {}).get('F', 16.0))
        
        # Results from solution
        self.a = getattr(solution_instance, 'a', 
                        getattr(solution_instance, 'results', {}).get('a', None))
        self.F_N = getattr(solution_instance, 'F_N', 
                          getattr(solution_instance, 'results', {}).get('F_N', None))
        
        # Store original values for perturbation tests
        self._original_mB = self.mB
        self._original_F = self.F
        
        # Expected values from physics
        self.expected_a = self.F / (self.mA + self.mB) if self.F and self.mA and self.mB else None
        self.expected_F_N = self.mB * self.expected_a if self.expected_a and self.mB else None

    def test_invariants(self) -> List[str]:
        logs = []
        
        try:
            # Newton's 2nd Law for system: F = (mA + mB)*a
            if self.a is not None and self.F is not None:
                total_force = (self.mA + self.mB) * self.a
                if math.isclose(self.F, total_force, rel_tol=0.02, abs_tol=0.02):
                    logs.append("✓ System Newton's 2nd Law satisfied: F ≈ (mA + mB)·a")
                else:
                    logs.append(f"✗ System Newton's 2nd Law violated: {self.F} ≠ ({self.mA}+{self.mB})·{self.a} = {total_force}")
            
            # Newton's 2nd Law for object B: F_N = mB * a
            if self.F_N is not None and self.a is not None and self.mB is not None:
                force_on_B = self.mB * self.a
                if math.isclose(self.F_N, force_on_B, rel_tol=0.02, abs_tol=0.02):
                    logs.append("✓ Newton's 2nd Law for object B satisfied: F_N ≈ mB·a")
                else:
                    logs.append(f"✗ Newton's 2nd Law for object B violated: {self.F_N} ≠ {self.mB}·{self.a} = {force_on_B}")
            
            # Energy conservation check (kinetic energy) - PURELY NUMERIC
            if self.a is not None and self.F is not None:
                # For uniformly accelerated motion starting from rest
                # Choose an arbitrary time t = 1.0 s for numeric comparison
                t = 1.0
                v = self.a * t
                d = 0.5 * self.a * t**2
                work = self.F * d
                m_total = self.mA + self.mB
                ke = 0.5 * m_total * v**2
                
                # Using F = m_total * a (should make them equal)
                work_expected = (m_total * self.a) * d
                
                if math.isclose(work, work_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append("✓ Energy conservation satisfied for constant acceleration")
                else:
                    logs.append(f"✗ Energy conservation check failed: work={work}, work_expected={work_expected}")
                    
        except Exception as e:
            logs.append(f"✗ Error in test_invariants: {traceback.format_exc()}")
        
        return logs

    def test_trace_consistency(self) -> List[str]:
        logs = []
        
        try:
            # Check if intermediate acceleration matches expectation
            if self.a is not None and self.expected_a is not None:
                if math.isclose(self.a, self.expected_a, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✓ Acceleration consistent: a = F/(mA+mB) = {self.a:.3f} m/s²")
                else:
                    logs.append(f"✗ Acceleration mismatch: got {self.a}, expected {self.expected_a:.3f}")
            
            # Check if final answer matches 6N
            if self.F_N is not None:
                if math.isclose(self.F_N, 6.0, rel_tol=0.02, abs_tol=0.02):
                    logs.append("✓ Final answer matches target: F_N ≈ 6N")
                else:
                    logs.append(f"✗ Final answer mismatch: got {self.F_N}N, expected 6N")
            
            # Cross-check with step-by-step calculation
            if self.a is not None and self.F_N is not None and self.mB is not None:
                recalculated_F_N = self.mB * self.a
                if math.isclose(self.F_N, recalculated_F_N, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"✓ Step consistency: F_N = mB·a = {self.mB}·{self.a:.3f} = {self.F_N:.3f}N")
                else:
                    logs.append(f"✗ Step inconsistency: mB·a = {recalculated_F_N:.3f} ≠ reported F_N = {self.F_N:.3f}")
                    
        except Exception as e:
            logs.append(f"✗ Error in test_trace_consistency: {traceback.format_exc()}")
        
        return logs

    def test_monotonicity(self) -> List[str]:
        logs = []
        
        try:
            # Perturb mB by +10% and check effect on F_N
            original_mB = self.mB
            perturbed_mB = original_mB * 1.10
            
            # Recalculate using physics formulas - PURELY NUMERIC
            perturbed_a = self.F / (self.mA + perturbed_mB)
            perturbed_F_N = perturbed_mB * perturbed_a
            
            # Check trends
            if self.expected_a is not None and self.expected_F_N is not None:
                # a should decrease with increased mass
                if perturbed_a < self.expected_a:
                    logs.append(f"✓ Acceleration decreases with increased mB: {perturbed_a:.3f} < {self.expected_a:.3f}")
                else:
                    logs.append(f"✗ Acceleration monotonicity violation: {perturbed_a:.3f} ≥ {self.expected_a:.3f}")
                
                # F_N = mB * F/(mA+mB) = F * mB/(mA+mB)
                # Derivative w.r.t mB: F*mA/(mA+mB)² > 0, so F_N should increase
                if perturbed_F_N > self.expected_F_N:
                    logs.append(f"✓ Contact force increases with increased mB: {perturbed_F_N:.3f} > {self.expected_F_N:.3f}")
                else:
                    logs.append(f"✗ Contact force monotonicity violation: {perturbed_F_N:.3f} ≤ {self.expected_F_N:.3f}")
            
            # Perturb F by +10% and check effects
            original_F = self.F
            perturbed_F = original_F * 1.10
            
            perturbed_a2 = perturbed_F / (self.mA + self.mB)
            perturbed_F_N2 = self.mB * perturbed_a2
            
            # Both a and F_N should increase proportionally with F
            if perturbed_a2 > self.expected_a and perturbed_F_N2 > self.expected_F_N:
                logs.append(f"✓ Both acceleration and contact force increase with increased F")
            else:
                logs.append(f"✗ Force perturbation shows unexpected trend")
                
            # Restore original values
            self.mB = original_mB
            self.F = original_F
            
        except Exception as e:
            logs.append(f"✗ Error in test_monotonicity: {traceback.format_exc()}")
            # Ensure restoration even on error
            self.mB = self._original_mB
            self.F = self._original_F
        
        return logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        success = True
        
        all_logs.append("=== Physics Checker Report ===")
        all_logs.append(f"Inputs: mA={self.mA}kg, mB={self.mB}kg, F={self.F}N")
        all_logs.append(f"Results: a={self.a}, F_N={self.F_N}")
        all_logs.append("")
        
        # Run individual tests
        test_functions = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_functions:
            all_logs.append(f"--- {test_name} ---")
            test_logs = test_func()
            all_logs.extend(test_logs)
            
            # Check for failures in this test
            for log in test_logs:
                if log.startswith("✗") or "Error" in log:
                    success = False
            all_logs.append("")
        
        # Overall assessment
        if success:
            all_logs.append("✓ All physics checks passed")
        else:
            all_logs.append("✗ Some physics checks failed")
        
        return success, all_logs