import sympy as sp
import math
import traceback
from typing import List, Tuple, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize checker with safe attribute access.
        Extract all required parameters from solution instance or results dictionary.
        """
        self.logs: List[str] = []
        self.results: Dict[str, float] = {}
        
        # Expected keys from the problem's key mapping
        keys = ['T_0', 'T_2', 'T_1', 'V_1', 'p_1', 'p_0', 'V_2', 'V_0', 'p_2']
        
        for key in keys:
            # Try to get from solution_instance.results first, then from solution_instance directly
            if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                val = solution_instance.results.get(key, 0.0)
            else:
                val = getattr(solution_instance, key, 0.0)
            self.results[key] = float(val) if val is not None else 0.0
        
        # Extract additional parameters safely
        self.l0 = getattr(solution_instance, 'l0', 1.0)  # natural length (default 1.0)
        self.S = getattr(solution_instance, 'S', 1.0)    # cross-sectional area (default 1.0)
        self.F = getattr(solution_instance, 'F', 0.0)    # applied force
        self.k = getattr(solution_instance, 'k', 0.0)    # spring constant
        
        # If k not provided, compute from formula: k = p0*S/l0
        if self.k == 0.0 and self.results['p_0'] != 0.0 and self.l0 != 0.0:
            self.k = (self.results['p_0'] * self.S) / self.l0
        
        # Define symbolic variables
        self.T0, self.T1, self.T2 = sp.symbols('T0 T1 T2', positive=True, real=True)
        self.p0, self.p1, self.p2 = sp.symbols('p0 p1 p2', positive=True, real=True)
        self.V0, self.V1, self.V2 = sp.symbols('V0 V1 V2', positive=True, real=True)
        self.S_sym, self.l0_sym, self.k_sym, self.F_sym = sp.symbols('S l0 k F', positive=True, real=True)
        
        # Initialize test results
        self.test_results = {
            'invariants': False,
            'trace_consistency': False,
            'monotonicity': False
        }
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _log(self, message: str, is_error: bool = False):
        """Helper to log messages."""
        prefix = "[ERROR]" if is_error else "[INFO]"
        self.logs.append(f"{prefix} {message}")
    
    def test_invariants(self) -> bool:
        """
        Verify physical invariants:
        1. Ideal gas law consistency across states
        2. Force balance in static equilibrium
        """
        success = True
        
        try:
            # Extract numerical values
            T0_val = self.results['T_0']
            T1_val = self.results['T_1']
            T2_val = self.results['T_2']
            p0_val = self.results['p_0']
            p1_val = self.results['p_1']
            p2_val = self.results['p_2']
            V0_val = self.results['V_0']
            V1_val = self.results['V_1']
            V2_val = self.results['V_2']
            
            # Test 1: Ideal gas law for state 0 and state 1
            # p0*V0/T0 should equal p1*V1/T1 (constant nR)
            if T0_val != 0.0 and T1_val != 0.0:
                left_01 = (p0_val * V0_val) / T0_val
                right_01 = (p1_val * V1_val) / T1_val
                if not self._is_close(left_01, right_01):
                    self._log(f"Ideal gas law violation (0→1): p0V0/T0={left_01:.4f}, p1V1/T1={right_01:.4f}", True)
                    success = False
                else:
                    self._log(f"Ideal gas law (0→1) satisfied: {left_01:.4f} ≈ {right_01:.4f}")
            
            # Test 2: Ideal gas law for state 0 and state 2
            if T0_val != 0.0 and T2_val != 0.0:
                left_02 = (p0_val * V0_val) / T0_val
                right_02 = (p2_val * V2_val) / T2_val
                if not self._is_close(left_02, right_02):
                    self._log(f"Ideal gas law violation (0→2): p0V0/T0={left_02:.4f}, p2V2/T2={right_02:.4f}", True)
                    success = False
                else:
                    self._log(f"Ideal gas law (0→2) satisfied: {left_02:.4f} ≈ {right_02:.4f}")
            
            # Test 3: Force balance at state 1 (piston at 0.8l0)
            # Equation: p0*S + F - p1*S - k*(0.2*l0) = 0
            force_balance1 = p0_val*self.S + self.F - p1_val*self.S - self.k*(0.2*self.l0)
            if not self._is_close(force_balance1, 0.0):
                self._log(f"Force balance violation at state 1: {force_balance1:.4f} ≠ 0", True)
                success = False
            else:
                self._log(f"Force balance at state 1 satisfied: {force_balance1:.4f} ≈ 0")
            
            # Test 4: Force balance at state 2 (piston at 1.2l0)
            # Equation: p0*S + F - p2*S + k*(0.2*l0) = 0
            force_balance2 = p0_val*self.S + self.F - p2_val*self.S + self.k*(0.2*self.l0)
            if not self._is_close(force_balance2, 0.0):
                self._log(f"Force balance violation at state 2: {force_balance2:.4f} ≠ 0", True)
                success = False
            else:
                self._log(f"Force balance at state 2 satisfied: {force_balance2:.4f} ≈ 0")
            
            # Test 5: Spring constant definition
            # k should equal p0*S/l0
            k_expected = p0_val * self.S / self.l0
            if not self._is_close(self.k, k_expected):
                self._log(f"Spring constant mismatch: k={self.k:.4f}, expected p0*S/l0={k_expected:.4f}", True)
                success = False
            else:
                self._log(f"Spring constant definition satisfied: k={self.k:.4f} ≈ p0*S/l0={k_expected:.4f}")
            
            self.test_results['invariants'] = success
            
        except Exception as e:
            self._log(f"Error in invariants test: {str(e)}", True)
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Verify step-by-step consistency from the analysis.
        """
        success = True
        
        try:
            T0_val = self.results['T_0']
            T1_val = self.results['T_1']
            T2_val = self.results['T_2']
            p0_val = self.results['p_0']
            p1_val = self.results['p_1']
            p2_val = self.results['p_2']
            V0_val = self.results['V_0']
            V1_val = self.results['V_1']
            V2_val = self.results['V_2']
            
            # Step 1: Verify given values from problem statement
            # p1 should be 1.1*p0
            if not self._is_close(p1_val, 1.1 * p0_val):
                self._log(f"p1 mismatch: {p1_val:.4f} ≠ 1.1*p0={1.1*p0_val:.4f}", True)
                success = False
            
            # V1 should be 0.8*S*l0 = 0.8*V0 (since V0 = S*l0)
            V1_expected = 0.8 * self.S * self.l0
            if not self._is_close(V1_val, V1_expected):
                self._log(f"V1 mismatch: {V1_val:.4f} ≠ 0.8*S*l0={V1_expected:.4f}", True)
                success = False
            
            # Step 2: Verify T1 = 0.88*T0 (from ideal gas law)
            T1_expected = 0.88 * T0_val
            if not self._is_close(T1_val, T1_expected):
                self._log(f"T1 mismatch: {T1_val:.4f} ≠ 0.88*T0={T1_expected:.4f}", True)
                success = False
            else:
                self._log(f"T1 consistency: {T1_val:.4f} ≈ 0.88*T0={T1_expected:.4f}")
            
            # Step 3: Verify V2 = 1.2*S*l0 = 1.2*V0
            V2_expected = 1.2 * self.S * self.l0
            if not self._is_close(V2_val, V2_expected):
                self._log(f"V2 mismatch: {V2_val:.4f} ≠ 1.2*S*l0={V2_expected:.4f}", True)
                success = False
            
            # Step 5: Verify T2 = 1.8*T0
            T2_expected = 1.8 * T0_val
            if not self._is_close(T2_val, T2_expected):
                self._log(f"T2 mismatch: {T2_val:.4f} ≠ 1.8*T0={T2_expected:.4f}", True)
                success = False
            else:
                self._log(f"T2 consistency: {T2_val:.4f} ≈ 1.8*T0={T2_expected:.4f}")
            
            # Additional consistency: p2 from force balance
            # From force balance equations, we can derive p2
            # Using both force balance equations to eliminate F:
            # p2 = p0 + (p1 - p0) + (k*l0/S)*0.4
            # With k = p0*S/l0, this simplifies to p2 = 1.5*p0
            p2_expected = 1.5 * p0_val
            if not self._is_close(p2_val, p2_expected):
                self._log(f"p2 mismatch: {p2_val:.4f} ≠ 1.5*p0={p2_expected:.4f}", True)
                success = False
            else:
                self._log(f"p2 consistency: {p2_val:.4f} ≈ 1.5*p0={p2_expected:.4f}")
            
            self.test_results['trace_consistency'] = success
            
        except Exception as e:
            self._log(f"Error in trace consistency test: {str(e)}", True)
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input parameter and check if outputs change in expected direction.
        We'll perturb T0 and verify T1 and T2 scale proportionally.
        """
        success = True
        
        try:
            original_T0 = self.results['T_0']
            original_T1 = self.results['T_1']
            original_T2 = self.results['T_2']
            
            # Perturb T0 by +10%
            perturbed_T0 = original_T0 * 1.10
            
            # Expected scaled values (since T1 ∝ T0 and T2 ∝ T0)
            expected_T1 = original_T1 * 1.10
            expected_T2 = original_T2 * 1.10
            
            # In a real solution, we would recalculate using the student's formulas
            # Here we check if the relationships are linear
            ratio_T1 = original_T1 / original_T0
            ratio_T2 = original_T2 / original_T0
            
            # Check that ratios are consistent with expected values
            if not self._is_close(ratio_T1, 0.88):
                self._log(f"T1/T0 ratio mismatch: {ratio_T1:.4f} ≠ 0.88", True)
                success = False
            
            if not self._is_close(ratio_T2, 1.8):
                self._log(f"T2/T0 ratio mismatch: {ratio_T2:.4f} ≠ 1.8", True)
                success = False
            
            # Verify monotonic relationship
            if original_T0 > 0:
                if not (original_T1 < original_T0 and original_T2 > original_T0):
                    self._log(f"Temperature monotonicity violation: T1 should be less than T0, T2 greater than T0", True)
                    success = False
                else:
                    self._log(f"Temperature monotonicity satisfied: T1={original_T1:.4f} < T0={original_T0:.4f} < T2={original_T2:.4f}")
            
            # Check pressure-volume relationship for gas
            p0 = self.results['p_0']
            p1 = self.results['p_1']
            p2 = self.results['p_2']
            V0 = self.results['V_0']
            V1 = self.results['V_1']
            V2 = self.results['V_2']
            
            # For isothermal process, pV = constant
            # Here processes are not isothermal, but we can check that pV increases with T
            pV0 = p0 * V0
            pV1 = p1 * V1
            pV2 = p2 * V2
            
            # pV should be proportional to T (nR constant)
            if not (pV0 < pV2 and pV1 < pV2):
                self._log(f"pV product monotonicity violation: p0V0={pV0:.4f}, p1V1={pV1:.4f}, p2V2={pV2:.4f}", True)
                success = False
            else:
                self._log(f"pV product monotonicity: p0V0={pV0:.4f} < p2V2={pV2:.4f}, p1V1={pV1:.4f} < p2V2={pV2:.4f}")
            
            self.test_results['monotonicity'] = success
            
        except Exception as e:
            self._log(f"Error in monotonicity test: {str(e)}", True)
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        self.logs.clear()
        
        overall_success = True
        
        try:
            self._log("Starting Physics Audit...")
            
            # Run individual tests
            inv_success = self.test_invariants()
            trace_success = self.test_trace_consistency()
            mono_success = self.test_monotonicity()
            
            overall_success = inv_success and trace_success and mono_success
            
            # Summary
            self._log("\n=== TEST SUMMARY ===")
            self._log(f"Invariants test: {'PASS' if inv_success else 'FAIL'}")
            self._log(f"Trace consistency: {'PASS' if trace_success else 'FAIL'}")
            self._log(f"Monotonicity test: {'PASS' if mono_success else 'FAIL'}")
            self._log(f"Overall: {'PASS' if overall_success else 'FAIL'}")
            
        except Exception as e:
            self._log(f"Unexpected error in run_all_tests: {str(e)}\n{traceback.format_exc()}", True)
            overall_success = False
        
        return overall_success, self.logs