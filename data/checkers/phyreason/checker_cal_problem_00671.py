import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction using getattr with fallback to results dict
        self.P_rated = 66.0  # Given in problem
        self.R = 2.0         # Internal resistance (given)
        self.U = 220.0       # Operating voltage (given)
        
        # Extract results from solution with safe defaults
        self.I_normal = getattr(solution_instance, 'I', solution_instance.results.get('I', 0.0))
        self.P_heat = getattr(solution_instance, 'P_heat', solution_instance.results.get('P_heat', 0.0))
        self.P_motor = getattr(solution_instance, 'P_motor', solution_instance.results.get('P_motor', 0.0))
        self.I_stalled = getattr(solution_instance, 'I_stalled', solution_instance.results.get('I_stalled', 0.0))
        self.P_stalled = getattr(solution_instance, 'P_stalled', solution_instance.results.get('P_stalled', 0.0))
        
        # Expected values from steps analysis (for trace consistency)
        self.expected_values = {
            'I_normal': 0.3,
            'P_motor': 65.82,
            'I_stalled': 110.0,
            'P_stalled': 24200.0
        }
        
        self.logs = []
        
    def _log(self, message: str):
        """Helper to log messages"""
        self.logs.append(message)
        
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust fuzzy comparison"""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Test fundamental physics laws"""
        success = True
        
        # 1. Normal operation: P_rated = U * I_normal (from step 1)
        expected_I = self.P_rated / self.U
        if not self._is_close(self.I_normal, expected_I):
            self._log(f"FAIL: Normal current mismatch. Calculated {expected_I:.3f}, got {self.I_normal:.3f}")
            success = False
        else:
            self._log(f"PASS: Normal current I = {self.I_normal:.3f}A matches P_rated/U")
        
        # 2. Heat power from Joule's Law: P_heat = I_normal² * R
        expected_P_heat = self.I_normal**2 * self.R
        if not self._is_close(self.P_heat, expected_P_heat):
            self._log(f"FAIL: Heat power mismatch. Calculated {expected_P_heat:.3f}W, got {self.P_heat:.3f}W")
            success = False
        else:
            self._log(f"PASS: Heat power {self.P_heat:.3f}W matches I²R")
        
        # 3. Motor power: P_motor = P_rated - P_heat (from step 3)
        expected_P_motor = self.P_rated - self.P_heat
        if not self._is_close(self.P_motor, expected_P_motor):
            self._log(f"FAIL: Motor power mismatch. Calculated {expected_P_motor:.3f}W, got {self.P_motor:.3f}W")
            success = False
        else:
            self._log(f"PASS: Motor power {self.P_motor:.3f}W matches P_rated - P_heat")
        
        # 4. Stalled current: I_stalled = U / R (from step 4, Ohm's Law for pure resistor)
        expected_I_stalled = self.U / self.R
        if not self._is_close(self.I_stalled, expected_I_stalled):
            self._log(f"FAIL: Stalled current mismatch. Calculated {expected_I_stalled:.1f}A, got {self.I_stalled:.1f}A")
            success = False
        else:
            self._log(f"PASS: Stalled current {self.I_stalled:.1f}A matches U/R")
        
        # 5. Stalled power: P_stalled = U² / R (from step 5)
        expected_P_stalled = self.U**2 / self.R
        if not self._is_close(self.P_stalled, expected_P_stalled):
            self._log(f"FAIL: Stalled power mismatch. Calculated {expected_P_stalled:.1f}W, got {self.P_stalled:.1f}W")
            success = False
        else:
            self._log(f"PASS: Stalled power {self.P_stalled:.1f}W matches U²/R")
        
        # 6. Alternative check: P_stalled = U * I_stalled
        alt_P_stalled = self.U * self.I_stalled
        if not self._is_close(self.P_stalled, alt_P_stalled):
            self._log(f"FAIL: Stalled power inconsistent. U*I_stalled = {alt_P_stalled:.1f}W ≠ {self.P_stalled:.1f}W")
            success = False
        else:
            self._log(f"PASS: Stalled power consistent with U*I_stalled")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check intermediate results match expected values from steps"""
        success = True
        
        for key, expected in self.expected_values.items():
            actual = getattr(self, key)
            if not self._is_close(actual, expected):
                self._log(f"TRACE FAIL: {key} = {actual} differs from expected {expected}")
                success = False
            else:
                self._log(f"TRACE PASS: {key} matches expected value")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb resistance and check trends.
        Returns to original state after test.
        """
        success = True
        
        # Save original state
        original_R = self.R
        
        # Test 1: Increase resistance by 10%
        self.R = original_R * 1.1
        
        # Recalculate expected values with perturbed R
        # Normal operation: I_normal should remain constant (P_rated/U independent of R)
        I_normal_new = self.P_rated / self.U
        if not self._is_close(I_normal_new, self.I_normal):
            self._log(f"MONOTONICITY FAIL: I_normal changed with R (from {self.I_normal} to {I_normal_new})")
            success = False
        
        # P_heat = I²R should increase with R
        P_heat_new = self.I_normal**2 * self.R
        if P_heat_new <= self.P_heat:
            self._log(f"MONOTONICITY FAIL: P_heat did not increase with R (from {self.P_heat} to {P_heat_new})")
            success = False
        
        # P_motor = P_rated - P_heat should decrease
        P_motor_new = self.P_rated - P_heat_new
        if P_motor_new >= self.P_motor:
            self._log(f"MONOTONICITY FAIL: P_motor did not decrease with R (from {self.P_motor} to {P_motor_new})")
            success = False
        
        # Stalled operation: I_stalled = U/R should decrease
        I_stalled_new = self.U / self.R
        if I_stalled_new >= self.I_stalled:
            self._log(f"MONOTONICITY FAIL: I_stalled did not decrease with R (from {self.I_stalled} to {I_stalled_new})")
            success = False
        
        # Stalled power: P_stalled = U²/R should decrease
        P_stalled_new = self.U**2 / self.R
        if P_stalled_new >= self.P_stalled:
            self._log(f"MONOTONICITY FAIL: P_stalled did not decrease with R (from {self.P_stalled} to {P_stalled_new})")
            success = False
        
        # Restore original R
        self.R = original_R
        
        if success:
            self._log("MONOTONICITY PASS: All trends correct with increased resistance")
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all checks and return success status with logs"""
        self.logs = []
        
        self._log("=== Physics Checker Started ===")
        
        # Run individual tests
        inv_success = self.test_invariants()
        trace_success = self.test_trace_consistency()
        mono_success = self.test_monotonicity()
        
        overall_success = inv_success and trace_success and mono_success
        
        self._log(f"\n=== Summary ===")
        self._log(f"Invariants: {'PASS' if inv_success else 'FAIL'}")
        self._log(f"Trace Consistency: {'PASS' if trace_success else 'FAIL'}")
        self._log(f"Monotonicity: {'PASS' if mono_success else 'FAIL'}")
        self._log(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs