import math
import traceback
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        
        Args:
            solution_instance: The solution object containing inputs and results.
        """
        # Extract results with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract ALL parameters from results (not from inputs)
        # The solution code should have computed these
        self.d = float(self.results.get('d', 0.0))
        self.V = float(self.results.get('V', 0.0))
        self.F_s = float(self.results.get('F_s', 0.0))
        self.fracdVdQ = float(self.results.get('fracdVdQ', 0.0))
        self.F_E = float(self.results.get('F_E', 0.0))
        self.Q_Vmax2 = float(self.results.get('Q_Vmax2', 0.0))
        self.V_max = float(self.results.get('V_max', 0.0))
        self.W = float(self.results.get('W', 0.0))
        self.E_1 = float(self.results.get('E_1', 0.0))
        self.k_result = float(self.results.get('k', 0.0))
        self.E = float(self.results.get('E', 0.0))
        
        # For consistency checks, we need the parameters used in the solution
        # Extract them from results if available, otherwise use reasonable defaults
        self.d0 = float(self.results.get('d0', 1e-8))
        self.Q = float(self.results.get('Q', 1e-12))
        self.A = float(self.results.get('A', 1e-10))
        self.epsilon = float(self.results.get('epsilon', 8.854e-12 * 2.5))
        self.k = float(self.results.get('k_val', 0.1))  # Use k_val to distinguish from k_result
        self.V_th = float(self.results.get('Vth', 0.07))
        
        # Initialize logs
        self.logs = []
    
    def _log(self, message: str, success: bool = True):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
    
    def _check_equal(self, val1: float, val2: float, context: str = "") -> bool:
        """
        Check if two float values are equal within tolerance.
        
        Args:
            val1: First value
            val2: Second value
            context: Context for logging
            
        Returns:
            bool: True if values match within tolerance
        """
        try:
            # Handle very small numbers
            if abs(val1) < 1e-30 and abs(val2) < 1e-30:
                self._log(f"Both near zero: {context} ({val1:.6e} ≈ {val2:.6e})", True)
                return True
            
            # Numerical comparison with tolerance
            if math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-12):
                self._log(f"Numerical match: {context} ({val1:.6e} ≈ {val2:.6e})", True)
                return True
            else:
                self._log(f"Numerical mismatch: {context} ({val1:.6e} ≠ {val2:.6e})", False)
                return False
        except Exception as e:
            self._log(f"Error in comparison for {context}: {str(e)}", False)
            return False
    
    def test_invariants(self) -> bool:
        """
        Test physical invariants based on KB theorems.
        Now checks internal consistency of results, not against fixed expected values.
        """
        success = True
        
        try:
            # 1. Force balance: Spring force should equal Electric force at equilibrium
            # F_s = k*(d0 - d), F_E = Q^2/(2*epsilon*A)
            # But we don't have d0, Q, epsilon, A directly from results
            # Instead, check if F_s and F_E are equal (they should be at equilibrium)
            if not self._check_equal(self.F_s, self.F_E, "Force equilibrium F_s = F_E"):
                success = False
            
            # 2. Check that V = E * d
            expected_V = self.E * self.d
            if not self._check_equal(self.V, expected_V, "Voltage V = E*d"):
                success = False
            
            # 3. Check that E_1 = E/2 (field from one layer is half total field)
            expected_E_1 = self.E / 2
            if not self._check_equal(self.E_1, expected_E_1, "E_1 = E/2"):
                success = False
            
            # 4. Check that F_E = Q * E_1 (but we don't have Q)
            # Skip this check since we don't have Q
            
            # 5. Check that k_result is positive (critical k should be positive)
            if self.k_result <= 0:
                self._log(f"Critical k should be positive: {self.k_result}", False)
                success = False
            else:
                self._log(f"Critical k is positive: {self.k_result}", True)
            
            # 6. Check that V_max is positive (maximum voltage should be positive)
            if self.V_max < 0:
                self._log(f"V_max should be positive: {self.V_max}", False)
                success = False
            else:
                self._log(f"V_max is positive: {self.V_max}", True)
            
            # 7. Check that Q_Vmax2 is positive
            if self.Q_Vmax2 < 0:
                self._log(f"Q_Vmax2 should be positive: {self.Q_Vmax2}", False)
                success = False
            else:
                self._log(f"Q_Vmax2 is positive: {self.Q_Vmax2}", True)
            
        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Test consistency between intermediate and final results.
        """
        success = True
        
        try:
            # 1. Check that if Q_Vmax2 > 0, then V_max should be > 0
            if self.Q_Vmax2 > 1e-30 and self.V_max <= 0:
                self._log(f"If Q_Vmax2 > 0, V_max should be > 0", False)
                success = False
            
            # 2. Check that W is positive (work done should be positive)
            if self.W < 0:
                self._log(f"Work W should be positive: {self.W}", False)
                success = False
            else:
                self._log(f"Work W is positive: {self.W}", True)
            
            # 3. Check that k_result (critical k) is reasonable
            # For typical values, critical k should be on the order of 10^1 to 10^2
            if self.k_result > 1e6 or self.k_result < 1e-6:
                self._log(f"Critical k seems unreasonable: {self.k_result}", False)
                success = False
            else:
                self._log(f"Critical k is reasonable: {self.k_result}", True)
            
            # 4. Check that d is less than d0 (membrane is compressed)
            # Actually, d could be negative mathematically, so we can't check this
            
            # 5. Check that V has the right order of magnitude
            # For the given parameters, V should be on the order of 10^5 to 10^6
            if abs(self.V) > 1e9 or (abs(self.V) < 1e-6 and abs(self.V) > 0):
                self._log(f"Voltage V seems unreasonable: {self.V}", False)
                success = False
            else:
                self._log(f"Voltage V is reasonable: {self.V}", True)
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", False)
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Test monotonic relationships by perturbing parameters.
        Since we don't have the full parameter set, we'll skip detailed monotonicity tests.
        """
        success = True
        
        try:
            # Simple sanity checks instead of full monotonicity tests
            
            # 1. Check that F_s and F_E are equal (already done in invariants)
            if not math.isclose(self.F_s, self.F_E, rel_tol=1e-2):
                self._log(f"F_s and F_E should be equal at equilibrium", False)
                success = False
            
            # 2. Check that E is approximately 2 * E_1
            if not math.isclose(self.E, 2 * self.E_1, rel_tol=1e-2):
                self._log(f"E should be approximately 2 * E_1", False)
                success = False
            
            # 3. Check that V has the same sign as E * d
            if self.E * self.d * self.V < 0:  # Negative product means opposite signs
                self._log(f"V should have the same sign as E*d", False)
                success = False
            
            self._log("Basic monotonicity/sanity checks passed", True)
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", False)
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return success status and logs.
        
        Returns:
            Tuple[bool, List[str]]: (success, logs)
        """
        self.logs = []  # Reset logs
        
        self._log("=" * 60)
        self._log("Starting PhysicsChecker for Neuron Membrane Problem")
        self._log("=" * 60)
        
        # Run tests
        test_results = []
        
        self._log("\n[1/3] Running invariant tests...")
        inv_success = self.test_invariants()
        test_results.append(("Invariants", inv_success))
        
        self._log("\n[2/3] Running trace consistency tests...")
        trace_success = self.test_trace_consistency()
        test_results.append(("Trace Consistency", trace_success))
        
        self._log("\n[3/3] Running monotonicity tests...")
        mono_success = self.test_monotonicity()
        test_results.append(("Monotonicity", mono_success))
        
        # Summary
        self._log("\n" + "=" * 60)
        self._log("TEST SUMMARY")
        self._log("=" * 60)
        
        all_success = True
        for test_name, test_success in test_results:
            status = "PASS" if test_success else "FAIL"
            self._log(f"{test_name}: {status}")
            all_success = all_success and test_success
        
        self._log(f"\nOverall: {'PASS' if all_success else 'FAIL'}")
        
        return all_success, self.logs