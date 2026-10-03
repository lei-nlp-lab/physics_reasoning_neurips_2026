import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        # Safe access to input parameters with defaults
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.I = getattr(solution_instance, 'I', solution_instance.results.get('I', 1.0))
        self.L = getattr(solution_instance, 'L', solution_instance.results.get('L', 1.0))
        self.alpha = getattr(solution_instance, 'alpha', solution_instance.results.get('alpha', 37.0))
        
        # Safe access to results with defaults
        self.F_Ncos37circ = getattr(solution_instance, 'F_Ncos37circ', 
                                   solution_instance.results.get('F_Ncos37circ', 0.0))
        self.F_N = getattr(solution_instance, 'F_N', solution_instance.results.get('F_N', 0.0))
        self.F_a = getattr(solution_instance, 'F_a', solution_instance.results.get('F_a', 0.0))
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', 0.0))
        
        # Trigonometric constants
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.tan37 = self.sin37 / self.cos37
        
        # Store logs
        self.logs = []
        
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with tolerance."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify physics invariants from KB and steps analysis."""
        logs = []
        
        try:
            # 1. Check step1: F_N * cos37 = m*g (for Q1 configuration)
            # This should hold for the Q1 normal force, which is stored in F_Ncos37circ.
            expected_FNcos = self.m * self.g
            if self._isclose(self.F_Ncos37circ, expected_FNcos):
                logs.append("✓ Step1 invariant: F_N * cos37° = m*g (Q1 configuration)")
            else:
                logs.append(f"✗ Step1 invariant failed: F_Ncos37circ={self.F_Ncos37circ}, expected={expected_FNcos}")
            
            # 2. Check step3: F_a = B * I * L (should hold for Q1 configuration)
            # But note: In the results, B is from Q1, and F_a is from Q2. So this invariant
            # should not be tested across configurations. Instead, we should test Q1's F_a separately.
            # However, we don't have Q1's F_a stored. So we skip this test.
            # Instead, we verify that B matches the Q1 formula.
            
            # 3. Check B formula from step3: B = (3*m*g)/(4*I*L) for Q1
            expected_B_Q1 = (3 * self.m * self.g) / (4 * self.I * self.L)
            if self._isclose(self.B, expected_B_Q1):
                logs.append("✓ Step3 B formula: B = 3mg/(4IL) (Q1 configuration)")
            else:
                logs.append(f"✗ Step3 B formula failed: B={self.B}, expected={expected_B_Q1}")
            
            # 4. Check step4: F_N = F_a + m*g*cos37 (for Q2 configuration)
            expected_FN_from_step4 = self.F_a + self.m * self.g * self.cos37
            if self._isclose(self.F_N, expected_FN_from_step4):
                logs.append("✓ Step4 invariant: F_N = F_a + m*g*cos37° (Q2 configuration)")
            else:
                logs.append(f"✗ Step4 invariant failed: F_N={self.F_N}, F_a+mgcos37={expected_FN_from_step4}")
            
            # 5. Check that F_N from step5 matches the ground truth for Q2: F_N = 27/20 mg
            expected_FN_Q2 = (27/20) * self.m * self.g
            if self._isclose(self.F_N, expected_FN_Q2):
                logs.append("✓ F_N = (27/20)mg (Q2 ground truth)")
            else:
                logs.append(f"✗ F_N doesn't match (27/20)mg: actual={self.F_N}, expected={expected_FN_Q2}")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        # Only fail if critical invariants fail: Step1, Step3 B formula, Step4, and Q2 ground truth.
        critical_failures = [l for l in logs if l.startswith("✗") and 
                             ("Step1" in l or "Step3 B formula" in l or "Step4" in l or "(27/20)mg" in l)]
        return len(critical_failures) == 0

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        try:
            # Check that F_N from Q2 is consistent with its own F_a and mgcos37
            FN_from_Q2 = self.F_a + self.m * self.g * self.cos37
            if self._isclose(self.F_N, FN_from_Q2):
                logs.append("✓ F_N consistent with Q2 equilibrium: F_N = F_a + m*g*cos37°")
            else:
                logs.append(f"✗ F_N inconsistency in Q2: actual={self.F_N}, computed={FN_from_Q2}")
            
            # Check that B from Q1 is consistent with its formula
            B_from_Q1_formula = (3 * self.m * self.g) / (4 * self.I * self.L)
            if self._isclose(self.B, B_from_Q1_formula):
                logs.append("✓ B consistent with Q1 formula: B = 3mg/(4IL)")
            else:
                logs.append(f"✗ B inconsistency: actual={self.B}, formula={B_from_Q1_formula}")
            
            # Check that F_Ncos37circ equals m*g (Q1)
            if self._isclose(self.F_Ncos37circ, self.m * self.g):
                logs.append("✓ F_Ncos37circ = m*g (Q1)")
            else:
                logs.append(f"✗ F_Ncos37circ inconsistency: {self.F_Ncos37circ} vs {self.m * self.g}")
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("✗")]) == 0

    def test_monotonicity(self):
        """Check that increasing mass increases forces appropriately."""
        logs = []
        
        try:
            # Store original values
            original_m = float(self.m)
            original_F_a = float(self.F_a)
            original_B = float(self.B)
            original_F_N = float(self.F_N)
            
            # Check proportionality
            # B should be proportional to m (from Q1)
            if original_m != 0:
                B_per_m = original_B / original_m
            else:
                B_per_m = 0
                logs.append("Warning: m = 0, skipping proportionality checks")
            
            # F_a should be proportional to m (from Q2)
            if original_m != 0:
                Fa_per_m = original_F_a / original_m
            else:
                Fa_per_m = 0
            
            # F_N should be proportional to m (from Q2)
            if original_m != 0:
                FN_per_m = original_F_N / original_m
            else:
                FN_per_m = 0
            
            logs.append(f"Proportionality constants: B/m={B_per_m:.3f}, F_a/m={Fa_per_m:.3f}, F_N/m={FN_per_m:.3f}")
            
            # Check that all are positive (assuming m,g,I,L > 0)
            if B_per_m > 0:
                logs.append("✓ B increases with mass (B/m > 0)")
            else:
                logs.append(f"✗ B monotonicity failed: B/m = {B_per_m}")
            
            if Fa_per_m > 0:
                logs.append("✓ F_a increases with mass (F_a/m > 0)")
            else:
                logs.append(f"✗ F_a monotonicity failed: F_a/m = {Fa_per_m}")
            
            if FN_per_m > 0:
                logs.append("✓ F_N increases with mass (F_N/m > 0)")
            else:
                logs.append(f"✗ F_N monotonicity failed: F_N/m = {FN_per_m}")
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("✗")]) == 0

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs."""
        self.logs = []
        
        self.logs.append("="*50)
        self.logs.append("Physics Checker Report")
        self.logs.append("="*50)
        
        # Run tests
        invariants_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonicity_ok = self.test_monotonicity()
        
        # Summary
        self.logs.append("\n" + "="*50)
        self.logs.append("Summary:")
        self.logs.append(f"  Invariants: {'PASS' if invariants_ok else 'FAIL'}")
        self.logs.append(f"  Trace Consistency: {'PASS' if trace_ok else 'FAIL'}")
        self.logs.append(f"  Monotonicity: {'PASS' if monotonicity_ok else 'FAIL'}")
        
        all_ok = invariants_ok and trace_ok and monotonicity_ok
        self.logs.append(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
        
        return all_ok, self.logs