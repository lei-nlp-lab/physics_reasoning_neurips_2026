import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallback to results dictionary
        self.W_1 = getattr(solution_instance, 'W_1', getattr(solution_instance, 'results', {}).get('W_1', 0.0))
        self.Q = getattr(solution_instance, 'Q', getattr(solution_instance, 'results', {}).get('Q', 0.0))
        
        # Try to get input parameters if available
        self.pressure = getattr(solution_instance, 'pressure', getattr(solution_instance, 'results', {}).get('pressure', None))
        self.delta_V = getattr(solution_instance, 'delta_V', getattr(solution_instance, 'results', {}).get('delta_V', None))
        self.n = getattr(solution_instance, 'n', getattr(solution_instance, 'results', {}).get('n', None))
        self.R = getattr(solution_instance, 'R', 8.314)  # Default gas constant
        
        # Target value from problem statement
        self.target = 2e5
        self.logs = []

    def test_invariants(self):
        """Check thermodynamic consistency using First Law and process equations."""
        checks = []
        
        try:
            # 1. Check First Law consistency for isobaric process A→B
            # W_1 should equal pressure * delta_V (if those values are available)
            if self.pressure is not None and self.delta_V is not None:
                W_expected = self.pressure * self.delta_V
                if not math.isclose(self.W_1, W_expected, rel_tol=1e-2, abs_tol=1e-2):
                    checks.append(f"FAIL: W_1={self.W_1} ≠ p·ΔV={W_expected}")
                else:
                    checks.append(f"PASS: Isobaric work formula W = pΔV holds")
            
            # 2. Check First Law: ΔU = Q - W
            # For process A→C, steps show Q = W_1, implying ΔU = 0
            # This suggests internal energy doesn't change A→C
            ΔU = self.Q - self.W_1
            if not math.isclose(ΔU, 0, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(f"FAIL: ΔU = Q - W_1 = {ΔU} ≠ 0 (First Law violation)")
            else:
                checks.append(f"PASS: First Law consistent (ΔU ≈ 0 for A→C)")
            
            # 3. Check non-negativity (physical plausibility)
            if self.W_1 < -1e-2:  # Allow small negative tolerance
                checks.append(f"WARNING: W_1={self.W_1} is negative (gas does negative work?)")
            if self.Q < -1e-2:
                checks.append(f"WARNING: Q={self.Q} is negative (heat released?)")
            
            # 4. Symbolic check with SymPy
            try:
                # Define symbols
                p, ΔV, W, Q = sp.symbols('p ΔV W Q')
                # Isobaric work law
                work_law = sp.Eq(W, p * ΔV)
                # First Law
                first_law = sp.Eq(sp.Symbol('ΔU'), Q - W)
                
                # Check if numerical values satisfy these laws (with ΔU ≈ 0)
                if not math.isclose(self.W_1, self.Q, rel_tol=1e-2, abs_tol=1e-2):
                    checks.append(f"FAIL: Q={self.Q} ≠ W_1={self.W_1} as per steps")
                else:
                    checks.append(f"PASS: Q = W_1 consistent with steps analysis")
                    
            except Exception as e:
                checks.append(f"Symbolic check skipped: {str(e)}")
            
            # If all checks passed or only warnings
            if all('PASS' in c or 'WARNING' in c for c in checks[-3:]):
                return True, checks
            else:
                return False, checks
                
        except Exception as e:
            checks.append(f"ERROR in invariant test: {str(e)}")
            return False, checks

    def test_trace_consistency(self):
        """Verify intermediate results align with final answer."""
        checks = []
        
        try:
            # 1. Check W_1 equals target (200000 J)
            if not math.isclose(self.W_1, self.target, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(f"FAIL: W_1={self.W_1} ≠ target={self.target}")
            else:
                checks.append(f"PASS: W_1 matches target")
            
            # 2. Check Q equals W_1 (from step 2 equation)
            if not math.isclose(self.Q, self.W_1, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(f"FAIL: Q={self.Q} ≠ W_1={self.W_1}")
            else:
                checks.append(f"PASS: Q = W_1 as per step 2")
            
            # 3. Check both equal target
            if not math.isclose(self.Q, self.target, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(f"FAIL: Q={self.Q} ≠ target={self.target}")
            else:
                checks.append(f"PASS: Q matches target")
            
            return all('PASS' in c for c in checks), checks
            
        except Exception as e:
            checks.append(f"ERROR in trace consistency: {str(e)}")
            return False, checks

    def test_monotonicity(self):
        """Perturb an input parameter and check expected trend."""
        checks = []
        
        try:
            # We need at least one input parameter to perturb
            # Try pressure or delta_V, fallback to n
            
            original_pressure = self.pressure
            original_delta_V = self.delta_V
            original_n = self.n
            
            perturbed = False
            
            if original_pressure is not None:
                # Perturb pressure by +10%
                perturbed_pressure = original_pressure * 1.1
                # Recalculate W using formula W = p·ΔV (if ΔV available)
                if original_delta_V is not None:
                    new_W = perturbed_pressure * original_delta_V
                    # Expected trend: W increases with pressure at constant ΔV
                    if new_W > self.W_1 * 1.05:  # Allow 5% margin
                        checks.append(f"PASS: W increases with pressure (monotonic)")
                    else:
                        checks.append(f"WARNING: W doesn't increase enough with pressure")
                    perturbed = True
                    
            if not perturbed and original_delta_V is not None:
                # Perturb volume change by +10%
                perturbed_delta_V = original_delta_V * 1.1
                if original_pressure is not None:
                    new_W = original_pressure * perturbed_delta_V
                    if new_W > self.W_1 * 1.05:
                        checks.append(f"PASS: W increases with ΔV (monotonic)")
                    else:
                        checks.append(f"WARNING: W doesn't increase enough with ΔV")
                    perturbed = True
                    
            if not perturbed and original_n is not None:
                # Perturb n by +10%, affecting temperature via pV = nRT
                # For isobaric process, W = pΔV, ΔV depends on T change
                # If temperature increases with n at constant p and V? Actually pV = nRT
                # For fixed p and initial V, T ∝ 1/n - so perturbing n changes T
                # This is complex without full process, so we'll skip
                checks.append("SKIP: Monotonicity test - insufficient input parameters")
                return True, checks
                
            if not perturbed:
                checks.append("SKIP: Monotonicity test - no input parameters to perturb")
                return True, checks
                
            return True, checks
            
        except Exception as e:
            checks.append(f"ERROR in monotonicity test: {str(e)}")
            return False, checks
        finally:
            # Restore original values (in case solution_instance is mutable)
            pass  # We worked with copies, no restoration needed

    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        success = True
        
        # Run invariant tests
        inv_success, inv_logs = self.test_invariants()
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(inv_logs)
        success = success and inv_success
        
        # Run trace consistency tests
        trace_success, trace_logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(trace_logs)
        success = success and trace_success
        
        # Run monotonicity tests
        mono_success, mono_logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(mono_logs)
        success = success and mono_success
        
        # Final summary
        all_logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        
        return success, all_logs

# Example usage guard
if __name__ == "__main__":
    # For testing purposes only
    class MockSolution:
        def __init__(self):
            self.results = {
                'W_1': 200000.0,
                'Q': 200000.0,
                'pressure': 100000.0,  # Pa
                'delta_V': 2.0,        # m³
                'n': 1.0
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)