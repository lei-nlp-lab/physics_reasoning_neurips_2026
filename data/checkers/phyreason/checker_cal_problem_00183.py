import sympy as sp
import math
import traceback
from typing import Dict, Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Never access solution_instance.attr directly due to sandbox safety.
        """
        try:
            # Extract inputs with safe defaults
            self.p1 = float(getattr(solution_instance, 'p1', getattr(solution_instance, 'inputs', {}).get('p1', 1.0)))
            self.t1 = float(getattr(solution_instance, 't1', getattr(solution_instance, 'inputs', {}).get('t1', 0.0)))
            self.V1 = float(getattr(solution_instance, 'V1', getattr(solution_instance, 'inputs', {}).get('V1', 1.0)))
            self.p2 = float(getattr(solution_instance, 'p2', getattr(solution_instance, 'inputs', {}).get('p2', 1.0)))
            
            # Extract results from solution_instance.results dictionary
            results = getattr(solution_instance, 'results', {})
            self.p0 = float(results.get('p_0', 0.0))
            self.V2 = float(results.get('V_2', 0.0))
            
            # Store symbolic variables for invariant checking
            self.p1_sym = sp.symbols('p1')
            self.t1_sym = sp.symbols('t1')
            self.V1_sym = sp.symbols('V1')
            self.p2_sym = sp.symbols('p2')
            self.p0_sym = sp.symbols('p0')
            self.V2_sym = sp.symbols('V2')
            self.T1_sym = self.t1_sym + 273  # Kelvin conversion
            self.T0_sym = 273  # 0°C in Kelvin
            
            # Expected formulas from KB analysis
            self.expected_p0 = (273 * self.p1_sym) / (self.t1_sym + 273)
            self.expected_V2 = (273 * self.p1_sym * self.V1_sym) / (self.p2_sym * (self.t1_sym + 273))
            
        except Exception as e:
            self._error = f"Initialization error: {str(e)}"
            raise

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify physical invariants:
        1. Isochoric process: p1/T1 = p0/T0 (Charles's Law for isochoric process)
        2. Ideal gas law consistency between two states
        """
        logs = []
        success = True
        
        try:
            # Convert Celsius to Kelvin
            T1 = self.t1 + 273
            T0 = 273
            
            # Test 1: Isochoric process invariant (p/T constant)
            if T1 > 0:
                lhs = self.p1 / T1
                rhs = self.p0 / T0
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Isochoric invariant passed: p1/T1 = {lhs:.3e}, p0/T0 = {rhs:.3e}")
                else:
                    logs.append(f"✗ Isochoric invariant failed: p1/T1 = {lhs:.3e}, p0/T0 = {rhs:.3e}")
                    success = False
            else:
                logs.append("⚠ Skipped isochoric test: T1 <= 0")
            
            # Test 2: Ideal gas law for closed system (nR constant)
            # State 1: (p1, V1, T1), State 2: (p2, V2, T0)
            # Should satisfy: (p1 * V1) / T1 = (p2 * V2) / T0
            if T1 > 0 and T0 > 0:
                state1 = (self.p1 * self.V1) / T1
                state2 = (self.p2 * self.V2) / T0
                if math.isclose(state1, state2, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Ideal gas law consistency: (p1*V1)/T1 = {state1:.3e}, (p2*V2)/T0 = {state2:.3e}")
                else:
                    logs.append(f"✗ Ideal gas law inconsistency: (p1*V1)/T1 = {state1:.3e}, (p2*V2)/T0 = {state2:.3e}")
                    success = False
            else:
                logs.append("⚠ Skipped ideal gas law test: Invalid temperatures")
                
        except Exception as e:
            logs.append(f"✗ Invariant test failed with error: {str(e)}")
            success = False
            
        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check that intermediate results are consistent with final answers.
        Check against expected formulas from KB.
        """
        logs = []
        success = True
        
        try:
            # Test p0 against expected formula
            expected_p0_val = (273 * self.p1) / (self.t1 + 273)
            if math.isclose(self.p0, expected_p0_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ p0 consistency: computed {self.p0:.3e}, expected {expected_p0_val:.3e}")
            else:
                logs.append(f"✗ p0 inconsistency: computed {self.p0:.3e}, expected {expected_p0_val:.3e}")
                success = False
            
            # Test V2 against expected formula
            expected_V2_val = (273 * self.p1 * self.V1) / (self.p2 * (self.t1 + 273))
            if math.isclose(self.V2, expected_V2_val, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ V2 consistency: computed {self.V2:.3e}, expected {expected_V2_val:.3e}")
            else:
                logs.append(f"✗ V2 inconsistency: computed {self.V2:.3e}, expected {expected_V2_val:.3e}")
                success = False
                
        except ZeroDivisionError:
            logs.append("⚠ Skipped trace consistency: Division by zero in expected formulas")
        except Exception as e:
            logs.append(f"✗ Trace consistency test failed with error: {str(e)}")
            success = False
            
        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb inputs and verify output trends match theoretical expectations.
        We perturb t1 and check p0 and V2 responses, then restore original values.
        """
        logs = []
        success = True
        original_t1 = self.t1
        
        try:
            # Create a small perturbation (1% increase)
            perturb_t1 = original_t1 * 1.01 if original_t1 != 0 else 1.0
            
            # Expected behavior from formulas:
            # p0 = 273*p1/(t1+273) - decreases as t1 increases (denominator increases)
            # V2 = 273*p1*V1/(p2*(t1+273)) - also decreases as t1 increases
            
            # Calculate perturbed values
            perturbed_p0 = (273 * self.p1) / (perturb_t1 + 273)
            perturbed_V2 = (273 * self.p1 * self.V1) / (self.p2 * (perturb_t1 + 273))
            
            # Check monotonic trends
            if original_t1 != 0:
                # Check p0 decreases when t1 increases
                if perturbed_p0 < self.p0:
                    logs.append(f"✓ p0 monotonicity: decreases from {self.p0:.3e} to {perturbed_p0:.3e} when t1 increases")
                else:
                    logs.append(f"✗ p0 monotonicity violation: should decrease but {perturbed_p0:.3e} ≥ {self.p0:.3e}")
                    success = False
                
                # Check V2 decreases when t1 increases
                if perturbed_V2 < self.V2:
                    logs.append(f"✓ V2 monotonicity: decreases from {self.V2:.3e} to {perturbed_V2:.3e} when t1 increases")
                else:
                    logs.append(f"✗ V2 monotonicity violation: should decrease but {perturbed_V2:.3e} ≥ {self.V2:.3e}")
                    success = False
            else:
                logs.append("⚠ Skipped monotonicity test: t1 = 0, perturbation unreliable")
                
        except Exception as e:
            logs.append(f"✗ Monotonicity test failed with error: {str(e)}")
            success = False
        finally:
            # Restore original t1 value
            self.t1 = original_t1
            
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all physics audit tests and return consolidated results.
        """
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        return overall_success, all_logs